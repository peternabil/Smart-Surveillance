# Generated by Django 3.2 on 2021-06-26 19:12

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('MainApp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Analytic',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('avg_persons_num', models.BigIntegerField()),
                ('avg_male', models.BigIntegerField()),
                ('avg_female', models.BigIntegerField()),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='MainApp.video')),
            ],
            options={
                'ordering': ['avg_persons_num'],
            },
        ),
    ]
